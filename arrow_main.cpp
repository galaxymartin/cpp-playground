#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <iostream>
#include <iterator>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

arrow::Status RunMain()
{
  arrow::Int8Builder int8builder;
  int8_t days_raw[5] = {1, 12, 17, 23, 28};
  // AppendValues, as called, puts 5 values from days_raw into our Builder object.
  ARROW_RETURN_NOT_OK(int8builder.AppendValues(days_raw, 5));

  // We only have a Builder though, not an Array -- the following code pushes out the
  // built up data into a proper Array.
  std::shared_ptr<arrow::Array> days;
  ARROW_ASSIGN_OR_RAISE(days, int8builder.Finish());

  // Builders clear their state every time they fill an Array, so if the type is the same,
  // we can re-use the builder. We do that here for month values.
  int8_t months_raw[5] = {1, 3, 5, 7, 1};
  ARROW_RETURN_NOT_OK(int8builder.AppendValues(months_raw, 5));
  std::shared_ptr<arrow::Array> months;
  ARROW_ASSIGN_OR_RAISE(months, int8builder.Finish());

  arrow::Int16Builder int16builder;
  int16_t years_raw[5] = {2000, 2010, 2020, 2030, 2040};
  ARROW_RETURN_NOT_OK(int16builder.AppendValues(years_raw, 5));
  std::shared_ptr<arrow::Array> years;
  ARROW_ASSIGN_OR_RAISE(years, int16builder.Finish());

  std::shared_ptr<arrow::Field> field_day, field_month, field_year;
  std::shared_ptr<arrow::Schema> schema;

  field_day = arrow::field("Day", arrow::int8());
  field_month = arrow::field("Month", arrow::int8());
  field_year = arrow::field("Year", arrow::int16());

  schema = arrow::schema({field_day, field_month, field_year});

  std::shared_ptr<arrow::RecordBatch> rbatch;
  rbatch = arrow::RecordBatch::Make(schema, days->length(), {days, months, years});

  std::cout << rbatch->ToString();

  int8_t days_raw2[5] = {6, 12, 3, 30, 22};
  ARROW_RETURN_NOT_OK(int8builder.AppendValues(days_raw2, 5));
  std::shared_ptr<arrow::Array> days2;
  ARROW_ASSIGN_OR_RAISE(days2, int8builder.Finish());

  int8_t months_raw2[5] = {5, 4, 11, 3, 2};
  ARROW_RETURN_NOT_OK(int8builder.AppendValues(months_raw2, 5));
  std::shared_ptr<arrow::Array> months2;
  ARROW_ASSIGN_OR_RAISE(months2, int8builder.Finish());

  int16_t years_raw2[5] = {1980, 2001, 1915, 2020, 1996};
  ARROW_RETURN_NOT_OK(int16builder.AppendValues(years_raw2, 5));
  std::shared_ptr<arrow::Array> years2;
  ARROW_ASSIGN_OR_RAISE(years2, int16builder.Finish());

  arrow::ArrayVector day_vec{days, days2};
  std::shared_ptr<arrow::ChunkedArray> day_chunks =
      std::make_shared<arrow::ChunkedArray>(day_vec);

  arrow::ArrayVector month_vecs{months, months2};
  std::shared_ptr<arrow::ChunkedArray> month_chunks =
      std::make_shared<arrow::ChunkedArray>(month_vecs);

  arrow::ArrayVector year_vecs{years, years2};
  std::shared_ptr<arrow::ChunkedArray> year_chunks =
      std::make_shared<arrow::ChunkedArray>(year_vecs);

  // Create a table
  std::shared_ptr<arrow::Table> table;
  table = arrow::Table::Make(schema, {day_chunks, month_chunks, year_chunks}, 10);

  std::cout << table->ToString();

  // Initialize the compute module to register the required compute kernels.
  ARROW_RETURN_NOT_OK(arrow::compute::Initialize());

  arrow::Datum sum;
  ARROW_ASSIGN_OR_RAISE(sum, arrow::compute::Sum({table->GetColumnByName("Day")}));

  if (sum.is_scalar()) {
     const std::shared_ptr<arrow::Scalar>& scalar_ptr = sum.scalar();
     if (scalar_ptr->is_valid) {
        // Safely cast to the concrete scalar type to access the value
        auto specific_scalar = std::static_pointer_cast<arrow::Int64Scalar>(scalar_ptr);
        std::cout << "Retrieved scalar value: " << specific_scalar->value << std::endl;
     }
  }  

  return arrow::Status::OK();
}

int main(int ac, char *av[])
{
  try
  {

    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("compression", po::value<double>(), "set compression level");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      std::cout << desc << "\n";
      return 0;
    }

    if (vm.count("compression"))
    {
      std::cout << "Compression level was set to "
                << vm["compression"].as<double>() << ".\n";
    }
    else
    {
      std::cout << "Compression level was not set.\n";
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "Exception of unknown type!\n";
  }

  arrow::Status st = RunMain();
  if (!st.ok())
  {
    std::cerr << st << std::endl;
    return 1;
  }
  return 0;
}
